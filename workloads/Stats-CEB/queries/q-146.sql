SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND p.Id = v.PostId
  AND p.Id = c.PostId
  AND p.Id = pl.PostId
  AND p.Id = ph.PostId
  AND c.CreationDate >= CAST('2010-07-26 19:37:03' AS timestamp)
  AND p.Score >= -2
  AND p.CommentCount <= 18
  AND p.CreationDate >= CAST('2010-07-21 13:50:08' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-11 00:53:10' AS timestamp)
  AND pl.CreationDate <= CAST('2014-08-05 18:27:51' AS timestamp)
  AND ph.CreationDate >= CAST('2010-11-27 03:38:45' AS timestamp)
  AND u.DownVotes >= 0
  AND u.UpVotes >= 0;
