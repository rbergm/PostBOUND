SELECT COUNT(*)
FROM posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  badges AS b,
  users AS u
WHERE p.Id = pl.RelatedPostId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND u.Id = ph.UserId
  AND u.Id = v.UserId
  AND p.CommentCount >= 0
  AND p.CommentCount <= 13
  AND ph.PostHistoryTypeId = 5
  AND ph.CreationDate <= CAST('2014-08-13 09:20:10' AS timestamp)
  AND v.CreationDate >= CAST('2010-07-19 00:00:00' AS timestamp)
  AND b.Date <= CAST('2014-09-09 10:24:35' AS timestamp)
  AND u.Views >= 0
  AND u.DownVotes >= 0
  AND u.CreationDate >= CAST('2010-08-04 16:59:53' AS timestamp)
  AND u.CreationDate <= CAST('2014-07-22 15:15:22' AS timestamp);
