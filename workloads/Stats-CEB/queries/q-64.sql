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
  AND c.CreationDate <= CAST('2014-09-11 13:24:22' AS timestamp)
  AND p.PostTypeId = 1
  AND p.Score = 2
  AND p.FavoriteCount <= 12
  AND pl.CreationDate >= CAST('2010-08-13 11:42:08' AS timestamp)
  AND pl.CreationDate <= CAST('2014-08-29 00:27:05' AS timestamp)
  AND ph.CreationDate >= CAST('2011-01-03 23:47:35' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-08 12:48:36' AS timestamp)
  AND v.CreationDate >= CAST('2010-07-27 00:00:00' AS timestamp)
  AND u.Reputation >= 1
  AND u.DownVotes >= 0;
