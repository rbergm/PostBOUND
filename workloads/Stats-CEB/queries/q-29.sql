SELECT COUNT(*)
FROM votes AS v,
  posts AS p,
  badges AS b,
  users AS u
WHERE p.Id = v.PostId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND v.CreationDate <= CAST('2014-09-12 00:00:00' AS timestamp)
  AND p.PostTypeId = 1
  AND p.Score >= -1
  AND p.FavoriteCount >= 0
  AND p.FavoriteCount <= 20
  AND b.Date >= CAST('2010-07-20 19:02:22' AS timestamp)
  AND b.Date <= CAST('2014-09-03 23:36:09' AS timestamp)
  AND u.DownVotes <= 2
  AND u.UpVotes >= 0
  AND u.CreationDate >= CAST('2010-11-26 03:34:11' AS timestamp);
