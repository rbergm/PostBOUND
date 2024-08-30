SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE p.Id = pl.PostId
  AND p.Id = ph.PostId
  AND p.Id = c.PostId
  AND u.Id = c.UserId
  AND u.Id = v.UserId
  AND c.CreationDate >= CAST('2010-07-26 20:21:15' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-13 01:26:16' AS timestamp)
  AND p.Score >= -1
  AND p.Score <= 19
  AND p.CommentCount <= 13
  AND ph.PostHistoryTypeId = 2
  AND ph.CreationDate <= CAST('2014-08-07 12:06:00' AS timestamp)
  AND v.BountyAmount <= 50
  AND v.CreationDate >= CAST('2010-07-21 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-14 00:00:00' AS timestamp)
  AND u.Views >= 0
  AND u.CreationDate <= CAST('2014-08-19 21:33:14' AS timestamp);
